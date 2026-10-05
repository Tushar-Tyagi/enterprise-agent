"""
Demonstration runner for the SQLite Company API and Enterprise Environment.
"""
from environment import SQLiteCompanyAPI, create_company_database


def main():
    print("==================================================")
    print(" Initializing Enterprise SQLite Environment")
    print("==================================================")

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")

    print(f"System Clock Initialized: {api.get_clock()}")
    user = api.get_user("u-101")
    print(f"Active User: {user['name']} ({user['title']}) | Max PO Limit: ${user['po_create_max_value']:,.2f}")
    print(f"Scopes: {', '.join(user['scopes'])}")

    # Scenario A: Handling shipment delay
    print("\n--- [Scenario A] Processing Mail Triggers & Expediting PO ---")
    emails = api.get_emails()
    print(f"Received {len(emails)} emails in inbox:")
    for m in emails:
        print(f"  [{m['mail_id']}] From: {m['sender']} | Subject: {m['subject']}")

    trigger = next(e for e in emails if e["mail_id"] == "M-001")
    print(f"\nProcessing Trigger: '{trigger['subject']}' -> {trigger['body']}")

    print("\nQuerying suppliers for part P-4471...")
    suppliers = api.query_suppliers(part_id="P-4471")
    for s in suppliers:
        status_str = "APPROVED" if s["approved"] else "UNAPPROVED (TRAP)"
        print(f"  * {s['name']} (ID: {s['supplier_id']}) - {status_str} - Lead Time: {s['lead_time_days']} days - ${s['unit_price']}/unit")

    # Select alternate
    print("\nSelecting valid approved alternate supplier with lead time <= 4 days...")
    approved = [s for s in suppliers if s["approved"] and s["lead_time_days"] <= 4]
    selected = approved[0]
    print(f"Selected: {selected['name']} ({selected['supplier_id']})")

    # Cancel old PO and issue new PO
    print("Cancelling delayed PO-77812...")
    api.cancel_po("PO-77812")
    print("Creating expedited PO-77815 with Supplier Z...")
    new_po = api.create_po(
        po_id="PO-77815",
        part_id="P-4471",
        supplier_id=selected["supplier_id"],
        quantity=50,
        unit_price=selected["unit_price"],
        promised_date="2026-09-04",
    )
    print(f"Created PO {new_po['po_id']}: ${new_po['total_amount']:,.2f}, Status: {new_po['status']}, Promised: {new_po['promised_date']}")

    # Scenario B: Quality hold reallocation
    print("\n--- [Scenario B] Quality Lot Hold Reallocation ---")
    with api.user_context("u-201"):  # Casey Chen (Quality Manager)
        lots = api.get_quality_lots(part_id="P-1180")
        print("Inspecting lots for part P-1180:")
        for lot in lots:
            alloc = f"(Allocated to Order {lot['allocated_order_id']})" if lot["allocated_order_id"] else "(Unallocated)"
            print(f"  * Lot {lot['lot_id']}: Status={lot['status']} {alloc} - Reason: {lot['hold_reason']}")

        print("\nReallocating available lot L-2094 to Order 4820...")
        realloc = api.reallocate_lot_for_order(
            order_id="4820",
            from_lot_id="L-2093",
            to_lot_id="L-2094",
        )
        print(f"Reallocation result: {realloc}")

        print("Notifying production supervisor Sam Taylor (u-301)...")
        notif = api.notify_production(
            supervisor_id="u-301",
            order_id="4820",
            message="Reallocated inspected lot L-2094 to Order 4820. L-2093 placed on hold.",
        )
        print(f"Notification sent: ID={notif['notification_id']} to {notif['supervisor_id']}")

    print("\n==================================================")
    print(" All scenarios executed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
